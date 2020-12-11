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
var StockEventsDirective = /** @class */ (function (_super) {
    __extends(StockEventsDirective, _super);
    function StockEventsDirective() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    StockEventsDirective.prototype.render = function () {
        return;
    };
    StockEventsDirective = __decorate([
        EJComponentDecorator({})
    ], StockEventsDirective);
    return StockEventsDirective;
}(Vue));
export { StockEventsDirective };
export var StockEventsPlugin = {
    name: 'e-stockchart-stockevents',
    install: function (Vue) {
        Vue.component(StockEventsPlugin.name, StockEventsDirective);
    }
};
var StockEventDirective = /** @class */ (function (_super) {
    __extends(StockEventDirective, _super);
    function StockEventDirective() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    StockEventDirective.prototype.render = function () {
        return;
    };
    StockEventDirective = __decorate([
        EJComponentDecorator({})
    ], StockEventDirective);
    return StockEventDirective;
}(Vue));
export { StockEventDirective };
export var StockEventPlugin = {
    name: 'e-stockchart-stockevent',
    install: function (Vue) {
        Vue.component(StockEventPlugin.name, StockEventDirective);
    }
};
