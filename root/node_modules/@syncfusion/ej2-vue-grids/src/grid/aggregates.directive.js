var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    }
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
import Vue from 'vue';
import { EJComponentDecorator } from '@syncfusion/ej2-vue-base';
var AggregatesDirective = /** @class */ (function (_super) {
    __extends(AggregatesDirective, _super);
    function AggregatesDirective() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    AggregatesDirective.prototype.render = function () {
        return;
    };
    AggregatesDirective = __decorate([
        EJComponentDecorator({})
    ], AggregatesDirective);
    return AggregatesDirective;
}(Vue));
export { AggregatesDirective };
export var AggregatesPlugin = {
    name: 'e-aggregates',
    install: function (Vue) {
        Vue.component(AggregatesPlugin.name, AggregatesDirective);
    }
};
/**
 * `e-aggregate` directive represent a aggregate row of the VueJS Grid.
 * It must be contained in a Grid component(`ejs-grid`).
 * ```vue
 * <ejs-grid :dataSource]='data' allowPaging='true' allowSorting='true'>
 *   <e-columns>
 *     <e-column field='ID' width='100'/>
 *     <e-column field='name' headerText='Name' width='100'/>
 *   </e-columns>
 *   <e-aggregates>
 *     <e-aggregate>
 *       <e-columns>
 *         <e-column field='ID' type='Min'/>
 *       </e-columns>
 *      </e-aggregate>
 *    </e-aggregates>
 * </ejs-grid>
 * ```
 */
var AggregateDirective = /** @class */ (function (_super) {
    __extends(AggregateDirective, _super);
    function AggregateDirective() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    AggregateDirective.prototype.render = function () {
        return;
    };
    AggregateDirective = __decorate([
        EJComponentDecorator({})
    ], AggregateDirective);
    return AggregateDirective;
}(Vue));
export { AggregateDirective };
export var AggregatePlugin = {
    name: 'e-aggregate',
    install: function (Vue) {
        Vue.component(AggregatePlugin.name, AggregateDirective);
    }
};
