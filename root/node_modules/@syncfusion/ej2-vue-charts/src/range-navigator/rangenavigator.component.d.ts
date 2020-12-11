import { ComponentBase } from '@syncfusion/ej2-vue-base';
export declare const properties: string[];
export declare const modelProps: string[];
/**
 * Represents Vuejs RangeNavigator Component
 * ```vue
 * <ejs-rangenavigator></ejs-rangenavigator>
 * ```
 */
export declare class RangeNavigatorComponent extends ComponentBase {
    ej2Instances: any;
    propKeys: string[];
    models: string[];
    hasChildDirective: boolean;
    protected hasInjectedModules: boolean;
    tagMapper: {
        [key: string]: Object;
    };
    tagNameMapper: Object;
    constructor();
    setProperties(prop: any, muteOnChange: boolean): void;
    trigger(eventName: string, eventProp: {
        [key: string]: Object;
    }, successHandler?: Function): void;
    render(createElement: any): any;
    createSecondaryElement(): void;
    export(type: Object, fileName: string, orientation?: Object, controls?: undefined[], width?: number, height?: number, isVertical?: boolean): void;
    print(id?: string[] | string | Object): void;
    renderChart(): void;
}
export declare const RangeNavigatorPlugin: {
    name: string;
    install(Vue: any): void;
};
